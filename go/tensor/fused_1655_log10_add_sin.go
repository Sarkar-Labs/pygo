package tensor

import "fmt"

func Fused1655Log10AddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Sin()
	return r, nil
}
func Fused1655Log10AddSinName() string { return "log10_add_sin" }
