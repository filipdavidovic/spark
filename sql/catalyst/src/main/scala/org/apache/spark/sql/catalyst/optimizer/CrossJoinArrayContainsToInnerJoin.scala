/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.catalyst.optimizer

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules._
import org.apache.spark.sql.catalyst.trees.TreePattern.{FILTER, JOIN}
import org.apache.spark.sql.types._

/**
 * Converts cross joins with array_contains filter into inner joins using explode.
 *
 * This optimization transforms queries of the form:
 * {{{
 * SELECT * FROM left, right WHERE array_contains(left.arr, right.elem)
 * }}}
 *
 * Into a more efficient form:
 * {{{
 * SELECT * FROM (
 *   SELECT *, explode(array_distinct(arr)) AS unnested FROM left
 * ) l
 * INNER JOIN right ON l.unnested = right.elem
 * }}}
 *
 * This avoids the O(N*M) cross join by using unnesting and equi-join.
 *
 * Ported from Presto's CrossJoinWithArrayContainsToInnerJoin optimizer rule.
 */
object CrossJoinArrayContainsToInnerJoin extends Rule[LogicalPlan] with PredicateHelper {

  // Supported element types for the optimization (matching Presto's supported types)
  private val supportedTypes: Set[DataType] = Set(
    IntegerType, LongType, StringType, DateType
  )

  override def apply(plan: LogicalPlan): LogicalPlan = plan.transformUpWithPruning(
    _.containsAllPatterns(FILTER, JOIN)) {
    case filter @ Filter(condition, join @ Join(left, right, Cross | Inner, None, _))
        if join.condition.isEmpty =>
      transformFilterOverCrossJoin(filter, condition, join, left, right)
        .getOrElse(filter)
  }

  /**
   * Attempts to transform a Filter over a Cross/Inner join (with no condition) that has
   * an array_contains predicate into an inner join with explode.
   */
  private def transformFilterOverCrossJoin(
      filter: Filter,
      condition: Expression,
      join: Join,
      left: LogicalPlan,
      right: LogicalPlan): Option[LogicalPlan] = {

    val conjuncts = splitConjunctivePredicates(condition)

    // Find an array_contains predicate that spans both sides
    val arrayContainsOpt = findArrayContainsPredicate(conjuncts, left, right)

    arrayContainsOpt.flatMap { case (arrayContains, arrayExpr, elementExpr, arrayOnLeft) =>
      // Get the remaining predicates (excluding the array_contains we're using)
      val remainingPredicates = conjuncts.filterNot(_ == arrayContains)

      // Build the transformation
      buildTransformedPlan(
        join, left, right, arrayExpr, elementExpr, arrayOnLeft, remainingPredicates)
    }
  }

  /**
   * Finds an array_contains predicate where the array comes from one side
   * and the element from the other side of the join.
   *
   * @return Option of (ArrayContains expression, array expression, element expression,
   *         true if array is on left side)
   */
  private def findArrayContainsPredicate(
      conjuncts: Seq[Expression],
      left: LogicalPlan,
      right: LogicalPlan): Option[(ArrayContains, Expression, Expression, Boolean)] = {

    val leftOutput = left.outputSet
    val rightOutput = right.outputSet

    conjuncts.collectFirst {
      case ac @ ArrayContains(arrayExpr, elementExpr)
          if isSupportedArrayContains(arrayExpr, elementExpr, leftOutput, rightOutput) =>

        val arrayOnLeft = arrayExpr.references.subsetOf(leftOutput)
        (ac, arrayExpr, elementExpr, arrayOnLeft)
    }
  }

  /**
   * Checks if the array_contains can be optimized:
   * - Element type is in supported types
   * - Array and element come from different sides of the join
   * - Both array and element are simple column references
   */
  private def isSupportedArrayContains(
      arrayExpr: Expression,
      elementExpr: Expression,
      leftOutput: AttributeSet,
      rightOutput: AttributeSet): Boolean = {

    // Check element type is supported
    val elementType = elementExpr.dataType
    val isTypeSupported = supportedTypes.contains(elementType) ||
      elementType.isInstanceOf[StringType]

    // Check that array is an array type with matching element type
    val arrayType = arrayExpr.dataType
    val isArrayTypeValid = arrayType match {
      case ArrayType(arrElemType, _) => arrElemType == elementType
      case _ => false
    }

    // Check that array comes from one side and element from the other
    val arrayRefs = arrayExpr.references
    val elemRefs = elementExpr.references

    val arrayFromLeft = arrayRefs.nonEmpty && arrayRefs.subsetOf(leftOutput)
    val arrayFromRight = arrayRefs.nonEmpty && arrayRefs.subsetOf(rightOutput)
    val elemFromLeft = elemRefs.nonEmpty && elemRefs.subsetOf(leftOutput)
    val elemFromRight = elemRefs.nonEmpty && elemRefs.subsetOf(rightOutput)

    val crossesSides = (arrayFromLeft && elemFromRight) || (arrayFromRight && elemFromLeft)

    isTypeSupported && isArrayTypeValid && crossesSides
  }

  /**
   * Builds the transformed plan with explode and inner join.
   */
  private def buildTransformedPlan(
      originalJoin: Join,
      left: LogicalPlan,
      right: LogicalPlan,
      arrayExpr: Expression,
      elementExpr: Expression,
      arrayOnLeft: Boolean,
      remainingPredicates: Seq[Expression]): Option[LogicalPlan] = {

    val elementType = elementExpr.dataType

    // Create array_distinct to avoid duplicate matches
    val distinctArray = ArrayDistinct(arrayExpr)

    // Create the explode generator
    val explodeExpr = Explode(distinctArray)

    // Create output attribute for the exploded values
    val unnestedAttr = AttributeReference("unnested", elementType, nullable = true)()

    // Determine which side has the array and create Generate node
    val (planWithGenerate: LogicalPlan, otherPlan: LogicalPlan, joinCondition: Expression) =
      if (arrayOnLeft) {
        val generate = Generate(
          generator = explodeExpr,
          unrequiredChildIndex = Nil,
          outer = false,
          qualifier = None,
          generatorOutput = Seq(unnestedAttr),
          child = left
        )
        val cond = EqualTo(unnestedAttr, elementExpr)
        (generate, right, cond)
      } else {
        val generate = Generate(
          generator = explodeExpr,
          unrequiredChildIndex = Nil,
          outer = false,
          qualifier = None,
          generatorOutput = Seq(unnestedAttr),
          child = right
        )
        val cond = EqualTo(elementExpr, unnestedAttr)
        (left, generate, cond)
      }

    // Create the inner join with the equi-join condition
    val innerJoin = if (arrayOnLeft) {
      Join(planWithGenerate, otherPlan, Inner, Some(joinCondition), JoinHint.NONE)
    } else {
      Join(otherPlan, planWithGenerate, Inner, Some(joinCondition), JoinHint.NONE)
    }

    // Project to match original output (excluding the unnested column)
    val originalOutput = originalJoin.output
    val projectList = originalOutput.map(a => Alias(a, a.name)(a.exprId))

    val projected = Project(projectList, innerJoin)

    // Add remaining filter predicates if any
    val result = if (remainingPredicates.nonEmpty) {
      Filter(remainingPredicates.reduceLeft(And), projected)
    } else {
      projected
    }

    Some(result)
  }
}
