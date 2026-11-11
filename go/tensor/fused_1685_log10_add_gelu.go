package tensor

import "fmt"

func Fused1685Log10AddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.GELU()
	return r, nil
}
func Fused1685Log10AddGeluName() string { return "log10_add_gelu" }
