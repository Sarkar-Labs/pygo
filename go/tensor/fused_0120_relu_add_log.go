package tensor

import "fmt"

func Fused0120ReluAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Log()
	return r, nil
}
func Fused0120ReluAddLogName() string { return "relu_add_log" }
