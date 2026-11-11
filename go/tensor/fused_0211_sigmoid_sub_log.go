package tensor

import "fmt"

func Fused0211SigmoidSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Log()
	return r, nil
}
func Fused0211SigmoidSubLogName() string { return "sigmoid_sub_log" }
