package tensor

import "fmt"

func Fused0106ReluSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Abs()
	return r, nil
}
func Fused0106ReluSubAbsName() string { return "relu_sub_abs" }
