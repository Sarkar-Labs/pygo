package tensor

import "fmt"

func Fused0680LogAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Ceil()
	return r, nil
}
func Fused0680LogAddCeilName() string { return "log_add_ceil" }
