package tensor

import "fmt"

func Fused0117ReluMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Exp()
	return r, nil
}
func Fused0117ReluMulExpName() string { return "relu_mul_exp" }
