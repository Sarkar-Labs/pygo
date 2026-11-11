package tensor

import "fmt"

func Fused0254SigmoidMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.SiLU()
	return r, nil
}
func Fused0254SigmoidMatmulSiluName() string { return "sigmoid_matmul_silu" }
