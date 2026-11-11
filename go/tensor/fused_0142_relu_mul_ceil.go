package tensor

import "fmt"

func Fused0142ReluMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Ceil()
	return r, nil
}
func Fused0142ReluMulCeilName() string { return "relu_mul_ceil" }
