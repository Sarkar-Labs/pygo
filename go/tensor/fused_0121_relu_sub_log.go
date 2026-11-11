package tensor

import "fmt"

func Fused0121ReluSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Log()
	return r, nil
}
func Fused0121ReluSubLogName() string { return "relu_sub_log" }
