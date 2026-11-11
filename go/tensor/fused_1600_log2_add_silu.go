package tensor

import "fmt"

func Fused1600Log2AddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.SiLU()
	return r, nil
}
func Fused1600Log2AddSiluName() string { return "log2_add_silu" }
