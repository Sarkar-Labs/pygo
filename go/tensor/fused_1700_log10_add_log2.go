package tensor

import "fmt"

func Fused1700Log10AddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Log2()
	return r, nil
}
func Fused1700Log10AddLog2Name() string { return "log10_add_log2" }
