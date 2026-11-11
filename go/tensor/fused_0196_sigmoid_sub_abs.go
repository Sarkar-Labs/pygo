package tensor

import "fmt"

func Fused0196SigmoidSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Abs()
	return r, nil
}
func Fused0196SigmoidSubAbsName() string { return "sigmoid_sub_abs" }
