package tensor

import "fmt"

func Fused0724SinMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.ReLU()
	return r, nil
}
func Fused0724SinMatmulReluName() string { return "sin_matmul_relu" }
