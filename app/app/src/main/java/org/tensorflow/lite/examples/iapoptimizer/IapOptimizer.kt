// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package org.tensorflow.lite.examples.iapoptimizer

import android.content.Context
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks.call
import com.google.firebase.analytics.FirebaseAnalytics
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class IapOptimizer(private val context: Context) {
  val testInput = mapOf(
    "coins_spent" to                       2048f,
    "distance_avg" to                      1234f,
    "device_os" to                         "ANDROID",
    "game_day" to                          10f,
    "geo_country" to                       "Canada",
    "last_run_end_reason" to               "laser"
  )

  private var interpreter: Interpreter? = null
  private var preprocessingSummary : JSONObject? = null

  var isInitialized = false
    private set

  /** Executor to run inference task in the background */
  private val executorService: ExecutorService = Executors.newCachedThreadPool()

  private var modelInputSize: Int = 0 // will be inferred from TF Lite model

  fun initialize(model: File): Task<Void> {
    return call(
      executorService,
      Callable<Void> {
        initializeInterpreter(model)
        null
      }
    )
  }

  private fun initializeInterpreter(model: File) {
    // Initialize TF Lite Interpreter with NNAPI enabled
    val options = Interpreter.Options()
    options.setUseNNAPI(true)
    val interpreter: Interpreter

    interpreter = Interpreter(model, options)

    val aFile = RandomAccessFile(model, "r")

    val inChannel: FileChannel = aFile.getChannel()
    val fileSize: Long = inChannel.size()

    val buffer = ByteBuffer.allocate(fileSize.toInt())
    inChannel.read(buffer)
    buffer.flip()

    val metaExecutor = MetadataExtractor(buffer)

    // Get associated preprocessing metadata JSON file from the TFLite file.
    // This is not yet supported on TFLite's iOS library, 
    // consider bundling this file separately instead.
    val inputStream = metaExecutor.getAssociatedFile("preprocess.json")
    val inputAsString = inputStream.bufferedReader().use { it.readText() }

    preprocessingSummary = JSONObject(inputAsString)

    // Read input shape from model file
    val inputShape = interpreter.getInputTensor(0).shape()

    modelInputSize = inputShape[1]

    // Finish interpreter initialization
    this.interpreter = interpreter
    isInitialized = true
    Log.d(TAG, "Initialized TFLite interpreter.")
  }

  fun predict(): String {
    if (!isInitialized) {
      throw IllegalStateException("TF Lite Interpreter is not initialized yet.")
    }

    val preprocessedInput = preprocessModelInput(testInput)
    val byteBuffer = Array(1) { preprocessedInput }

    val result = Array(1) { FloatArray(OUTPUT_ACTIONS_COUNT) }
    interpreter?.run(byteBuffer, result)

    return mapOutputToAction(result[0])
  }

  private fun preprocessModelInput(input : Map<String, Any>) : FloatArray {
    val result = mutableListOf<Float>()
    for ((k, v) in input) {
      if(!preprocessingSummary!!.has(k)) {
        continue
      }
      val channelSummary = preprocessingSummary!!.getJSONObject(k)

      if (channelSummary.getString("type") == "numerical") {
        val mean =channelSummary.getDouble("mean")
        val std =channelSummary.getDouble("std")
        if (v is Float) {
          result.add((v - mean.toFloat())/std.toFloat())
        } else {
          throw Exception("Invalid input")
        }
      }

      if (channelSummary.getString("type") == "categorical") {
        val allValues = channelSummary.getJSONArray("all_values")
        for (i in 0 until allValues.length()) {
          val possibleValue = allValues.getString(i)
          if (v == possibleValue) {
            result.add(1f)
          } else {
            result.add(0f)
          }
        }
      }
    }
    return result.toFloatArray()
  }

  private fun mapOutputToAction(output: FloatArray) : String {
    val mapping = preprocessingSummary!!.getJSONArray("output_mapping")

    val maxIndex = output.indexOf(output.max()!!)

    return mapping[maxIndex] as String
  }

  fun close() {
    call(
      executorService,
      Callable<String> {
        interpreter?.close()
        Log.d(TAG, "Closed TFLite interpreter.")
        null
      }
    )
  }

  companion object {
    private const val TAG = "IapOptimizer"

    private const val OUTPUT_ACTIONS_COUNT = 8
  }
}
