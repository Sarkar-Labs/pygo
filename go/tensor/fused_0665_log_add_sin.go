package tensor

import "fmt"

func Fused0665LogAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Sin()
	return r, nil
}
func Fused0665LogAddSinName() string { return "log_add_sin" }
