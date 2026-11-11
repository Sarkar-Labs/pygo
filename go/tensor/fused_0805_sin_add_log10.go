package tensor

import "fmt"

func Fused0805SinAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Log10()
	return r, nil
}
func Fused0805SinAddLog10Name() string { return "sin_add_log10" }
