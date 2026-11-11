package tensor

import "fmt"

func Fused1084RoundMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.ReLU()
	return r, nil
}
func Fused1084RoundMatmulReluName() string { return "round_matmul_relu" }
