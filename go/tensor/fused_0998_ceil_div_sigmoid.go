package tensor

import "fmt"

func Fused0998CeilDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Sigmoid()
	return r, nil
}
func Fused0998CeilDivSigmoidName() string { return "ceil_div_sigmoid" }
