//go:build examples
// +build examples

package main

import (
	"context"
	"fmt"
	"github.com/pygo/go/inference"
	"github.com/pygo/go/models"
	"github.com/pygo/go/tensor"
	"time"
)

func main() {
	fmt.Println("═══ PyGo LLaMA-3 Inference ═══")
	cfg := models.LLaMA3_8B()
	model := models.NewLLaMA(cfg)
	model.Train(false)
	total, _ := model.NumParams()
	fmt.Printf("%s\nParams: %.2fB\n", model, float64(total)/1e9)
	infCfg := inference.DefaultInferenceConfig()
	infCfg.Device = tensor.DevCPU
	infCfg.MaxNewTokens = 50
	predictor := inference.NewPredictor(model, infCfg)
	defer predictor.Close()
	inputIDs := []int64{1, 9906, 1917, 29892, 306, 626}
	fmt.Printf("Input: %v\n", inputIDs)
	start := time.Now()
	out, err := predictor.Greedy(context.Background(), inputIDs)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	if len(out.Sequences) > 0 {
		fmt.Printf("Generated: %v (%d tokens)\n", out.Sequences[0][len(inputIDs):], out.NumTokens)
	}
	fmt.Printf("Latency: %v  Speed: %.1f tok/s\n", time.Since(start).Round(time.Millisecond), float64(out.NumTokens)/time.Since(start).Seconds())
}
