package tensor

import "fmt"

func Fused1345GeluAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Log10()
	return r, nil
}
func Fused1345GeluAddLog10Name() string { return "gelu_add_log10" }
