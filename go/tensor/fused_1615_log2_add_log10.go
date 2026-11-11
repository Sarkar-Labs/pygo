package tensor

import "fmt"

func Fused1615Log2AddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Log10()
	return r, nil
}
func Fused1615Log2AddLog10Name() string { return "log2_add_log10" }
