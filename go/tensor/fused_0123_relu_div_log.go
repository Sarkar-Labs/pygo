package tensor

import "fmt"

func Fused0123ReluDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Log()
	return r, nil
}
func Fused0123ReluDivLogName() string { return "relu_div_log" }
