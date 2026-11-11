package tensor

import "fmt"

func Fused0631LogSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.ReLU()
	return r, nil
}
func Fused0631LogSubReluName() string { return "log_sub_relu" }
