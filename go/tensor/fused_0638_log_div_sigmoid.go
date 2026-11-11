package tensor

import "fmt"

func Fused0638LogDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Sigmoid()
	return r, nil
}
func Fused0638LogDivSigmoidName() string { return "log_div_sigmoid" }
