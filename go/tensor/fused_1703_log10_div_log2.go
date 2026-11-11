package tensor

import "fmt"

func Fused1703Log10DivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Log2()
	return r, nil
}
func Fused1703Log10DivLog2Name() string { return "log10_div_log2" }
