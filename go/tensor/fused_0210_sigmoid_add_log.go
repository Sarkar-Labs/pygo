package tensor

import "fmt"

func Fused0210SigmoidAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Log()
	return r, nil
}
func Fused0210SigmoidAddLogName() string { return "sigmoid_add_log" }
