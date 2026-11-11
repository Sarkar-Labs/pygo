package tensor

import "fmt"

func Fused1613Log2DivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Log2()
	return r, nil
}
func Fused1613Log2DivLog2Name() string { return "log2_div_log2" }
