package tensor

import "fmt"

func Fused0132ReluMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Cos()
	return r, nil
}
func Fused0132ReluMulCosName() string { return "relu_mul_cos" }
