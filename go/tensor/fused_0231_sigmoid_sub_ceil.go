package tensor

import "fmt"

func Fused0231SigmoidSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Ceil()
	return r, nil
}
func Fused0231SigmoidSubCeilName() string { return "sigmoid_sub_ceil" }
