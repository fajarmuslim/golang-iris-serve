package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
)

var model = tg.LoadModel("model", []string{"serve"}, nil)

// IrisRequest
type IrisRequest struct {
	SepalLength float32 `json:"sepal_length"`
	SepalWidth  float32 `json:"sepal_width"`
	PetalLength float32 `json:"petal_length"`
	PetalWidth  float32 `json:"petal_width"`
}

type IrisResponse struct {
	Result int `json:"result"`
}

// PostIrisRequest
func PostIrisRequest(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var irisRequest IrisRequest
	if r.Method == "POST" {
		if r.Header.Get("Content-Type") == "application/json" {
			// parse json
			decodeJSON := json.NewDecoder(r.Body)
			if err := decodeJSON.Decode(&irisRequest); err != nil {
				log.Fatal(err)
			}
		} else {
			// parse form
			getSepalLength := r.PostFormValue("sepalLength")
			sepalLength, _ := strconv.ParseFloat(getSepalLength, 64)

			getSepalWidth := r.PostFormValue("sepalWidth")
			sepalWidth, _ := strconv.ParseFloat(getSepalWidth, 64)

			getPetalLength := r.PostFormValue("sepalWidth")
			petalLength, _ := strconv.ParseFloat(getPetalLength, 64)

			getPetalWidth := r.PostFormValue("sepalWidth")
			petalWidth, _ := strconv.ParseFloat(getPetalWidth, 64)

			irisRequest = IrisRequest{
				SepalLength: float32(sepalLength),
				SepalWidth:  float32(sepalWidth),
				PetalLength: float32(petalLength),
				PetalWidth:  float32(petalWidth),
			}
		}

		irisInput, _ := tf.NewTensor([1][4]float32{{irisRequest.SepalLength, irisRequest.SepalWidth, irisRequest.PetalLength, irisRequest.PetalWidth}})
		results := model.Exec([]tf.Output{
			model.Op("StatefulPartitionedCall", 0),
		}, map[tf.Output]*tf.Tensor{
			model.Op("serving_default_dense_input", 0): irisInput,
		})

		predictions := results[0]
		for _, value := range predictions.Value().([][]float32) {

			idx_max := 0
			for idx, _ := range value {
				if value[idx] > value[idx_max] {
					idx_max = idx
				}
			}

			Hasil := IrisResponse{
				Result: idx_max,
			}

			dataIrisRequest, _ := json.Marshal(Hasil) // to byte
			w.Write(dataIrisRequest)                  // print in browser
		}

		return
	}

	http.Error(w, "hayo mau ngapain", http.StatusNotFound)
	return
}

func main() {
	http.HandleFunc("/iris", PostIrisRequest)
	fmt.Println("server running...")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
