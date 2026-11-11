package tensor

import "fmt"

func Fused0086Log10Sub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	return r, nil
}
func Fused0086Log10SubName() string { return "log10_sub" }
