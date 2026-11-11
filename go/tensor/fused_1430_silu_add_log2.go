package tensor

import "fmt"

func Fused1430SiluAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Log2()
	return r, nil
}
func Fused1430SiluAddLog2Name() string { return "silu_add_log2" }
