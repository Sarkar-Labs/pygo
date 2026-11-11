package tensor

import "fmt"

func Fused1435SiluAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Log10()
	return r, nil
}
func Fused1435SiluAddLog10Name() string { return "silu_add_log10" }
