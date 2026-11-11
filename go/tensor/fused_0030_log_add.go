package tensor

import "fmt"

func Fused0030LogAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	return r, nil
}
func Fused0030LogAddName() string { return "log_add" }
