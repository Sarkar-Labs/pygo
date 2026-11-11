package tensor

import "fmt"

func Fused0070SiluAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	return r, nil
}
func Fused0070SiluAddName() string { return "silu_add" }
