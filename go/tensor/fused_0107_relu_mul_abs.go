package tensor

import "fmt"

func Fused0107ReluMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Abs()
	return r, nil
}
func Fused0107ReluMulAbsName() string { return "relu_mul_abs" }
