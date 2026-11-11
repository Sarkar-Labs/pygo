package tensor

import "fmt"

func Fused1343GeluDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Log2()
	return r, nil
}
func Fused1343GeluDivLog2Name() string { return "gelu_div_log2" }
