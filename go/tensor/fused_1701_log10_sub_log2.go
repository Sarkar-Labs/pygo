package tensor

import "fmt"

func Fused1701Log10SubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Log2()
	return r, nil
}
func Fused1701Log10SubLog2Name() string { return "log10_sub_log2" }
