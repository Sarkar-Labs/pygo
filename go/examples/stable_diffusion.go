//go:build examples
// +build examples

package main

import (
	"fmt"
	"github.com/pygo/go/models"
	"github.com/pygo/go/tensor"
)

func main() {
	fmt.Println("═══ PyGo Stable Diffusion ═══")
	cfg := models.DDPMDefault()
	unet := models.NewUNetDDPM(cfg)
	fmt.Printf("UNet: T=%d C=%d\n", cfg.Timesteps, cfg.InChannels)
	x := tensor.Randn(tensor.Shape{1, int64(cfg.InChannels), 64, 64}, tensor.WithDevice(tensor.DevCPU))
	fmt.Printf("Noise: %v\n", x.Shape())
	for t := cfg.Timesteps - 1; t >= 0; t -= 100 {
		tT := tensor.FromSlice([]int64{int64(t)}, tensor.Shape{1})
		pred := unet.Forward(x, tT)
		alpha := float64(t) / float64(cfg.Timesteps)
		x = pred.MulScalar(alpha).Add(x.MulScalar(1 - alpha))
		fmt.Printf("  t=%4d\n", t)
	}
	fmt.Printf("Generated: %v\n", x.Shape())
}
