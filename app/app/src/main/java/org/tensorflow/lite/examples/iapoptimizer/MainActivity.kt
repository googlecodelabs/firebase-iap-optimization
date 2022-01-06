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

import android.annotation.SuppressLint
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.tasks.Task
import com.google.firebase.analytics.FirebaseAnalytics
import com.google.firebase.analytics.ktx.analytics
import com.google.firebase.analytics.ktx.logEvent
import com.google.firebase.ktx.Firebase
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel

private lateinit var firebaseAnalytics: FirebaseAnalytics

class MainActivity : AppCompatActivity() {

  private var predictButton: Button? = null
  private var acceptButton: Button? = null
  private var predictedTextView: TextView? = null
  private var iapOptimizer = IapOptimizer(this)
  private var predictionResult  = ""
  private var sessionId = "1"

  @SuppressLint("ClickableViewAccessibility")
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.tfe_dc_activity_main)

    // Obtain the FirebaseAnalytics instance.
    firebaseAnalytics = Firebase.analytics
    firebaseAnalytics.setUserId("player1")

    predictButton = findViewById(R.id.predict_button)
    acceptButton = findViewById(R.id.accept_button)
    predictedTextView = findViewById(R.id.predicted_text)
    predictedTextView?.text = "Click predict to see prediction result"

    predictButton?.setOnClickListener {
      val result = iapOptimizer.predict()
      predictedTextView?.text = "The best power-up to suggest: ${result}"
      predictionResult = result

      firebaseAnalytics.logEvent("offer_iap"){
        param("offer_type", predictionResult)
        param("offer_id", sessionId)
      }
    }

    acceptButton?.setOnClickListener {
      firebaseAnalytics.logEvent("offer_accepted") {
        param("offer_type", predictionResult)
        param("offer_id", sessionId)
      }
    }

    downloadModel("optimizer")
  }

  private fun downloadModel(modelName: String): Task<Void> {
    val remoteModel = FirebaseCustomRemoteModel.Builder(modelName).build()
    val firebaseModelManager = FirebaseModelManager.getInstance()
    return firebaseModelManager
      .isModelDownloaded(remoteModel)
      .continueWithTask { task ->
        // Create update condition if model is already downloaded, otherwise create download
        // condition.
        val conditions = if (task.result != null && task.result == true) {
          FirebaseModelDownloadConditions.Builder()
            .requireWifi()
            .build() // Update condition that requires wifi.
        } else {
          FirebaseModelDownloadConditions.Builder().build(); // Download condition.
        }
        firebaseModelManager.download(remoteModel, conditions)
      }
      .addOnSuccessListener {
        firebaseModelManager.getLatestModelFile(remoteModel)
          .addOnCompleteListener {
            val model = it.result
            if (model == null) {
              showToast("Failed to get model file.")
            } else {
              showToast("Downloaded remote model: $modelName")
              iapOptimizer.initialize(model)
            }
          }
      }
      .addOnFailureListener {
        showToast("Model download failed for $modelName, please check your connection.")
      }
  }

  override fun onDestroy() {
    iapOptimizer.close()
    super.onDestroy()
  }

  private fun showToast(text: String) {
    Toast.makeText(
            this,
            text,
            Toast.LENGTH_LONG
    ).show()
  }

  companion object {
    private const val TAG = "MainActivity"
  }
}
