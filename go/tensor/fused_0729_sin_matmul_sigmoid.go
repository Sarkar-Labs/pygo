package tensor

import "fmt"

func Fused0729SinMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Sigmoid()
	return r, nil
}
func Fused0729SinMatmulSigmoidName() string { return "sin_matmul_sigmoid" }
