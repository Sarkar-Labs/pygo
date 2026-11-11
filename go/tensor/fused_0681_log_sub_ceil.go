package tensor

import "fmt"

func Fused0681LogSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Ceil()
	return r, nil
}
func Fused0681LogSubCeilName() string { return "log_sub_ceil" }
