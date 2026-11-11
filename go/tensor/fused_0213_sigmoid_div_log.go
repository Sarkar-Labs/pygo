package tensor

import "fmt"

func Fused0213SigmoidDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Log()
	return r, nil
}
func Fused0213SigmoidDivLogName() string { return "sigmoid_div_log" }
