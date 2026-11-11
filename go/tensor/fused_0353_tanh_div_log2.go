package tensor

import "fmt"

func Fused0353TanhDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Log2()
	return r, nil
}
func Fused0353TanhDivLog2Name() string { return "tanh_div_log2" }
