package tensor

import "fmt"

func Fused0274TanhMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.ReLU()
	return r, nil
}
func Fused0274TanhMatmulReluName() string { return "tanh_matmul_relu" }
