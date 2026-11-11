package tensor

import "fmt"

func Fused1431SiluSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Log2()
	return r, nil
}
func Fused1431SiluSubLog2Name() string { return "silu_sub_log2" }
