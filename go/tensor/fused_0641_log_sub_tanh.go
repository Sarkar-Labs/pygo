package tensor

import "fmt"

func Fused0641LogSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Tanh()
	return r, nil
}
func Fused0641LogSubTanhName() string { return "log_sub_tanh" }
