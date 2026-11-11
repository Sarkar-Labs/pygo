package tensor

import "fmt"

func Fused1670Log10AddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Ceil()
	return r, nil
}
func Fused1670Log10AddCeilName() string { return "log10_add_ceil" }
