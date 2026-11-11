package tensor

import "fmt"

func Fused0640LogAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Tanh()
	return r, nil
}
func Fused0640LogAddTanhName() string { return "log_add_tanh" }
