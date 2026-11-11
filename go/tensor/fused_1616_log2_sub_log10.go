package tensor

import "fmt"

func Fused1616Log2SubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Log10()
	return r, nil
}
func Fused1616Log2SubLog10Name() string { return "log2_sub_log10" }
