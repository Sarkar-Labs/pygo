package tensor

import "fmt"

func Fused1433SiluDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Log2()
	return r, nil
}
func Fused1433SiluDivLog2Name() string { return "silu_div_log2" }
