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

package org.apache.spark.sql.execution.benchmark

import org.apache.spark.benchmark.Benchmark
import org.apache.spark.sql.functions._
import org.apache.spark.sql.internal.SQLConf

/**
 * Benchmark to measure performance improvement of CrossJoinArrayContainsToInnerJoin optimization.
 *
 * This benchmark compares:
 * 1. Cross join with array_contains filter (unoptimized)
 * 2. Inner join with explode (manually optimized / what the rule produces)
 *
 * To run this benchmark:
 * {{{
 *   1. without sbt:
 *      bin/spark-submit --class <this class>
 *        --jars <spark core test jar>,<spark catalyst test jar> <spark sql test jar>
 *   2. build/sbt "sql/Test/runMain <this class>"
 *   3. generate result:
 *      SPARK_GENERATE_BENCHMARK_FILES=1 build/sbt "sql/Test/runMain <this class>"
 *      Results will be written to
 *        "benchmarks/CrossJoinArrayContainsToInnerJoinBenchmark-results.txt".
 * }}}
 */
object CrossJoinArrayContainsToInnerJoinBenchmark extends SqlBasedBenchmark {

  import spark.implicits._

  private def crossJoinWithArrayContains(numOrders: Int, numItems: Int, arraySize: Int): Unit = {
    val benchmark = new Benchmark(
      s"Cross join with array_contains ($numOrders orders, $numItems items, array size $arraySize)",
      numOrders.toLong * numItems,
      output = output
    )

    // Create orders table with array of item IDs
    val orders = spark.range(numOrders)
      .selectExpr(
        "id as order_id",
        s"array_repeat(cast((id % $numItems) as int), $arraySize) as item_ids"
      )
      .cache()

    // Create items table
    val items = spark.range(numItems)
      .selectExpr("cast(id as int) as item_id", "concat('item_', id) as item_name")
      .cache()

    // Force caching
    orders.count()
    items.count()

    // Register as temp views for SQL queries
    orders.createOrReplaceTempView("orders")
    items.createOrReplaceTempView("items")

    benchmark.addCase("Cross join + array_contains filter (unoptimized)", numIters = 3) { _ =>
      // Disable the optimization to measure the true cross-join+filter baseline
      withSQLConf(
        SQLConf.CROSS_JOINS_ENABLED.key -> "true",
        SQLConf.OPTIMIZER_EXCLUDED_RULES.key ->
          "org.apache.spark.sql.catalyst.optimizer.CrossJoinArrayContainsToInnerJoin") {
        // This query would be a cross join with filter without optimization
        val df = spark.sql(
          """
            |SELECT /*+ BROADCAST(items) */ o.order_id, i.item_id, i.item_name
            |FROM orders o, items i
            |WHERE array_contains(o.item_ids, i.item_id)
          """.stripMargin)
        df.noop()
      }
    }

    benchmark.addCase("Inner join with explode (optimized equivalent)", numIters = 3) { _ =>
      // This is what the optimization produces - explode + inner join
      val df = spark.sql(
        """
          |SELECT o.order_id, i.item_id, i.item_name
          |FROM (
          |  SELECT order_id, explode(array_distinct(item_ids)) as unnested_id
          |  FROM orders
          |) o
          |INNER JOIN items i ON o.unnested_id = i.item_id
        """.stripMargin)
      df.noop()
    }

    benchmark.addCase("Inner join with explode (DataFrame API)", numIters = 3) { _ =>
      val ordersExploded = orders
        .withColumn("unnested_id", explode(array_distinct($"item_ids")))
        .select($"order_id", $"unnested_id")

      val df = ordersExploded.join(items, $"unnested_id" === $"item_id")
      df.noop()
    }

    benchmark.run()

    orders.unpersist()
    items.unpersist()
  }

  private def scalabilityBenchmark(): Unit = {
    val benchmark = new Benchmark(
      "Scalability: varying array sizes",
      1000000L,
      output = output
    )

    val numOrders = 10000
    val numItems = 1000

    Seq(1, 5, 10, 50).foreach { arraySize =>
      val orders = spark.range(numOrders)
        .selectExpr(
          "id as order_id",
          s"transform(sequence(0, $arraySize - 1), " +
            s"x -> cast((id + x) % $numItems as int)) as item_ids"
        )

      val items = spark.range(numItems)
        .selectExpr("cast(id as int) as item_id", "concat('item_', id) as item_name")

      orders.createOrReplaceTempView("orders_scale")
      items.createOrReplaceTempView("items_scale")

      benchmark.addCase(s"array_size=$arraySize with explode optimization", numIters = 3) { _ =>
        val df = spark.sql(
          """
            |SELECT o.order_id, i.item_id, i.item_name
            |FROM (
            |  SELECT order_id, explode(array_distinct(item_ids)) as unnested_id
            |  FROM orders_scale
            |) o
            |INNER JOIN items_scale i ON o.unnested_id = i.item_id
          """.stripMargin)
        df.noop()
      }
    }

    benchmark.run()
  }

  private def dataTypeBenchmark(): Unit = {
    val benchmark = new Benchmark(
      "Different data types in array",
      1000000L,
      output = output
    )

    val numRows = 10000
    val numLookup = 1000
    val arraySize = 10

    // Integer arrays
    benchmark.addCase("Integer array", numIters = 3) { _ =>
      val left = spark.range(numRows)
        .selectExpr("id", s"array_repeat(cast(id % $numLookup as int), $arraySize) as arr")
      val right = spark.range(numLookup).selectExpr("cast(id as int) as elem")

      val df = left
        .withColumn("unnested", explode(array_distinct($"arr")))
        .join(right, $"unnested" === $"elem")
      df.noop()
    }

    // Long arrays
    benchmark.addCase("Long array", numIters = 3) { _ =>
      val left = spark.range(numRows)
        .selectExpr("id", s"array_repeat(id % $numLookup, $arraySize) as arr")
      val right = spark.range(numLookup).selectExpr("id as elem")

      val df = left
        .withColumn("unnested", explode(array_distinct($"arr")))
        .join(right, $"unnested" === $"elem")
      df.noop()
    }

    // String arrays
    benchmark.addCase("String array", numIters = 3) { _ =>
      val left = spark.range(numRows)
        .selectExpr("id", s"array_repeat(concat('key_', id % $numLookup), $arraySize) as arr")
      val right = spark.range(numLookup).selectExpr("concat('key_', id) as elem")

      val df = left
        .withColumn("unnested", explode(array_distinct($"arr")))
        .join(right, $"unnested" === $"elem")
      df.noop()
    }

    benchmark.run()
  }

  override def runBenchmarkSuite(mainArgs: Array[String]): Unit = {
    runBenchmark("CrossJoinArrayContainsToInnerJoin Benchmark") {
      // Small scale test
      crossJoinWithArrayContains(numOrders = 1000, numItems = 100, arraySize = 5)

      // Medium scale test
      crossJoinWithArrayContains(numOrders = 10000, numItems = 1000, arraySize = 10)

      // Scalability test with varying array sizes
      scalabilityBenchmark()

      // Data type comparison
      dataTypeBenchmark()
    }
  }
}
