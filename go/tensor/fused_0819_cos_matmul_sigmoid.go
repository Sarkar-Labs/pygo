package tensor

import "fmt"

func Fused0819CosMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Sigmoid()
	return r, nil
}
func Fused0819CosMatmulSigmoidName() string { return "cos_matmul_sigmoid" }
